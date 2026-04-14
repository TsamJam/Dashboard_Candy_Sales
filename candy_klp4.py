"""
Customer Shipping Analysis - Dashboard
Kelompok 4 | Study Case 4: US Candy Distributor

Cara jalanin: streamlit run candy_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Candy Shipping Analysis",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    [data-testid="stMetricValue"] { font-size: 26px; }
    h1 { color: #2E86AB; }
    h2 { color: #1F2937; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; }
    .insight { background: #1b2a3d; border-left: 4px solid #2E86AB;
               padding: 12px 16px; border-radius: 4px; margin: 10px 0; }
    .warning { background: #FEF3C7; border-left: 4px solid #F59E0B;
               padding: 12px 16px; border-radius: 4px; margin: 10px 0; }
    .danger  { background: #FEE2E2; border-left: 4px solid #DC2626;
               padding: 12px 16px; border-radius: 4px; margin: 10px 0; }
    .success { background: #D1FAE5; border-left: 4px solid #10B981;
               padding: 12px 16px; border-radius: 4px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING - mengikuti alur notebook
# ============================================================
@st.cache_data
def load_data(data_dir="data"):
    factories_df = pd.read_csv(f"{data_dir}/Candy_Factories.csv")
    products_df  = pd.read_csv(f"{data_dir}/Candy_Products.csv")
    sales_df     = pd.read_csv(f"{data_dir}/Candy_Sales.csv")
    targets_df   = pd.read_csv(f"{data_dir}/Candy_Targets.csv")
    uszips_df    = pd.read_csv(f"{data_dir}/uszips.csv")

    # Merge sesuai notebook
    df = pd.merge(sales_df, products_df, on='Product ID', how='left', suffixes=('', '_prod'))
    df = pd.merge(df, factories_df, on='Factory', how='left')
    df['Postal Code'] = df['Postal Code'].astype(str).str.zfill(5)
    uszips_df['zip']  = uszips_df['zip'].astype(str).str.zfill(5)
    df = pd.merge(df, uszips_df, left_on='Postal Code', right_on='zip', how='left')
    df = pd.merge(df, targets_df, on='Division', how='left')

    # Tanggal + fix offset Ship Date
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date']  = pd.to_datetime(df['Ship Date'])
    offset = (df['Ship Date'] - df['Order Date']).dt.days.min()
    df['Ship Date'] = df['Ship Date'] - pd.Timedelta(days=offset)
    df['Shipping Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Year']      = df['Order Date'].dt.year
    df['YearMonth'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()

    # Haversine sama dengan notebook
    def calc_distance(lat1, lon1, lat2, lon2):
        R = 3958.8
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    df['Distance'] = calc_distance(df['Latitude'], df['Longitude'], df['lat'], df['lng'])
    df['Margin']   = df['Gross Profit'] / df['Sales']

    return df, factories_df, targets_df


try:
    df, factories_df, targets_df = load_data()
except FileNotFoundError:
    st.error("⚠️ File CSV tidak ditemukan. Pastikan folder `candy_data/` ada di lokasi yang sama dengan script.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🍬 Filter")
st.sidebar.markdown("---")

years = sorted(df["Year"].unique())
sel_years = st.sidebar.multiselect("Tahun", years, default=years)

divisions = sorted(df["Division"].unique())
sel_divs = st.sidebar.multiselect("Divisi", divisions, default=divisions)

regions = sorted(df["Region"].unique())
sel_regions = st.sidebar.multiselect("Region", regions, default=regions)

mask = (df["Year"].isin(sel_years) &
        df["Division"].isin(sel_divs) &
        df["Region"].isin(sel_regions))
fdf = df[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Baris terfilter", f"{len(fdf):,}")
st.sidebar.caption(f"dari {len(df):,} transaksi")
st.sidebar.markdown("---")
st.sidebar.markdown("**Kelompok 4**")
st.sidebar.caption("Study Case 4 — US Candy Distributor")


# ============================================================
# HEADER
# ============================================================
st.title("🍬 Customer Shipping Analysis")
st.markdown("**US National Candy Distributor** Sales, geospatial, dan optimasi rute pengiriman (2021–2024)")


# ============================================================
# KPI ROW
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Sales",   f"${fdf['Sales'].sum():,.0f}")
c2.metric("Gross Profit",  f"${fdf['Gross Profit'].sum():,.0f}")
c3.metric("Avg Margin",    f"{fdf['Margin'].mean()*100:.1f}%")
c4.metric("Avg Distance",  f"{fdf['Distance'].mean():.0f} mi")
c5.metric("Avg Ship Days", f"{fdf['Shipping Duration'].mean():.1f} hari")

st.markdown("---")


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📍 Geospatial",
    "🚚 Optimasi Rute",
    "📈 Tren Penjualan",
    "🎯 Target vs Aktual",
    "💰 Margin & Factory",
    "🔍 Korelasi",
])


# ------------------------------------------------------------
# TAB 1 — GEOSPATIAL
# ------------------------------------------------------------
with tab1:
    st.header("Analisis Geospatial: Jarak Pengiriman")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Distribusi Jarak Factory → Customer")
        fig_h = px.histogram(fdf.dropna(subset=['Distance']), x='Distance',
                             nbins=50, color_discrete_sequence=['#2E86AB'])
        fig_h.add_vline(x=fdf['Distance'].mean(), line_dash="dash",
                        line_color="red",
                        annotation_text=f"Mean: {fdf['Distance'].mean():.0f} mi")
        fig_h.update_layout(height=380, xaxis_title="Jarak (miles)",
                            yaxis_title="Jumlah Order", showlegend=False)
        st.plotly_chart(fig_h, use_container_width=True)

    with col_b:
        st.subheader("Sebaran Jarak per Factory")
        fig_b = px.box(fdf.dropna(subset=['Distance']),
                       x='Factory', y='Distance', color='Factory',
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig_b.update_layout(height=380, showlegend=False, xaxis_title="")
        fig_b.update_xaxes(tickangle=30)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown(f"""
    <div class="insight">
    Jarak rata-rata factory ke customer adalah <b>{fdf['Distance'].mean():.0f} miles</b>
    dengan median <b>{fdf['Distance'].median():.0f} miles</b>. Distribusinya menyebar ada
    pengiriman dekat (di bawah 500 mi) sampai lintas benua (di atas 2.000 mi). Setiap factory
    punya pola sendiri tergantung lokasi customer-nya.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Top 10 Rute: Paling Efisien vs Paling Tidak Efisien")

    eff_routes = (fdf.dropna(subset=['Distance'])
                  .groupby(['Factory', 'state_id'])['Distance']
                  .mean().nsmallest(10).reset_index())
    eff_routes['Route'] = eff_routes['Factory'] + ' → ' + eff_routes['state_id']

    ineff_routes = (fdf.dropna(subset=['Distance'])
                    .groupby(['Factory', 'state_id'])['Distance']
                    .mean().nlargest(10).reset_index())
    ineff_routes['Route'] = ineff_routes['Factory'] + ' → ' + ineff_routes['state_id']

    col_e, col_i = st.columns(2)
    with col_e:
        fig_eff = px.bar(eff_routes, x='Distance', y='Route', orientation='h',
                         color='Distance', color_continuous_scale='Greens_r',
                         text=eff_routes['Distance'].apply(lambda x: f"{x:.0f} mi"),
                         title="✅ 10 Rute Paling Efisien")
        fig_eff.update_layout(yaxis={'categoryorder': 'total descending'},
                              height=420, coloraxis_showscale=False)
        st.plotly_chart(fig_eff, use_container_width=True)

    with col_i:
        fig_ineff = px.bar(ineff_routes, x='Distance', y='Route', orientation='h',
                           color='Distance', color_continuous_scale='Reds',
                           text=ineff_routes['Distance'].apply(lambda x: f"{x:.0f} mi"),
                           title="❌ 10 Rute Paling Tidak Efisien")
        fig_ineff.update_layout(yaxis={'categoryorder': 'total ascending'},
                                height=420, coloraxis_showscale=False)
        st.plotly_chart(fig_ineff, use_container_width=True)

    st.markdown("""
    <div class="insight">
    Rute terpendek umumnya factory yang berdekatan dengan state tujuan.
    Rute terjauh terjadi waktu factory di pantai timur ngirim ke customer di pantai barat
    (atau sebaliknya). Ini target utama untuk optimasi.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🗺️ Peta Factory & Customer")

    geo = (fdf.dropna(subset=['lat', 'lng'])
           .groupby(['City', 'state_id', 'lat', 'lng'])
           .agg(Sales=('Sales', 'sum'), Orders=('Order ID', 'nunique'))
           .reset_index())

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lon=geo['lng'], lat=geo['lat'],
        text=geo['City'] + ', ' + geo['state_id'] + '<br>Sales: $' + geo['Sales'].round(0).astype(str),
        marker=dict(size=np.sqrt(geo['Sales']) * 0.5 + 3,
                    color=geo['Sales'], colorscale='Viridis',
                    showscale=True, colorbar=dict(title="Sales"),
                    opacity=0.6, line=dict(width=0.5, color='white')),
        name='Customer'
    ))
    fig_map.add_trace(go.Scattergeo(
        lon=factories_df['Longitude'], lat=factories_df['Latitude'],
        text=factories_df['Factory'],
        marker=dict(size=18, color='red', symbol='star',
                    line=dict(width=2, color='white')),
        name='Factory'
    ))
    fig_map.update_layout(geo=dict(scope='usa', showland=True, landcolor='#F3F4F6'),
                          height=550, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_map, use_container_width=True)


# ------------------------------------------------------------
# TAB 2 — OPTIMASI RUTE
# ------------------------------------------------------------
with tab2:
    st.header("🚚 Optimasi Rute & Rekomendasi Relokasi")
    st.caption("Untuk setiap order, dihitung factory mana yang sebenarnya paling dekat lalu identifikasi produk yang paling banyak \"dikirim dari factory yang salah\".")

    @st.cache_data
    def compute_optimal(_df_in):
        out = _df_in.dropna(subset=['lat', 'lng']).copy()

        def calc_d(lat1, lon1, lat2, lon2):
            R = 3958.8
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))

        # Vectorized — hitung jarak ke semua factory sekaligus, jauh lebih cepat dari apply
        all_dists = []
        for _, fac in factories_df.iterrows():
            d = calc_d(fac['Latitude'], fac['Longitude'],
                       out['lat'].values, out['lng'].values)
            all_dists.append(d)
        all_dists = np.array(all_dists)

        out['Min_Dist'] = all_dists.min(axis=0)
        out['Optimal_Factory'] = factories_df.iloc[all_dists.argmin(axis=0)]['Factory'].values
        out['Potential_Savings'] = out['Distance'] - out['Min_Dist']
        return out

    opt_df = compute_optimal(fdf)

    top_ineff = (opt_df.groupby(['Product Name', 'Factory', 'Optimal_Factory', 'Division'])
                 .agg({'Distance': 'sum', 'Min_Dist': 'sum',
                       'Potential_Savings': 'sum', 'Units': 'sum'})
                 .sort_values('Potential_Savings', ascending=False)
                 .head(10).reset_index())

    st.subheader("Perbandingan Jarak: Saat Ini vs Optimal")
    plot_data = top_ineff.melt(id_vars='Product Name',
                               value_vars=['Distance', 'Min_Dist'],
                               var_name='Metric', value_name='Total Miles')
    plot_data['Metric'] = plot_data['Metric'].map({
        'Distance': 'Jarak Saat Ini',
        'Min_Dist': 'Jarak Optimal'
    })

    fig_opt = px.bar(plot_data, y='Product Name', x='Total Miles',
                     color='Metric', barmode='group', orientation='h',
                     color_discrete_map={'Jarak Saat Ini': '#D62246',
                                         'Jarak Optimal': '#06A77D'})
    fig_opt.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_opt, use_container_width=True)

    st.subheader("📋 Tabel Rekomendasi Relokasi")
    rec = top_ineff.copy()
    rec['Savings %'] = (rec['Potential_Savings'] / rec['Distance'] * 100).round(1)
    rec_disp = rec[['Product Name', 'Factory', 'Optimal_Factory',
                    'Distance', 'Min_Dist', 'Potential_Savings',
                    'Savings %', 'Units']].rename(columns={
        'Factory': 'Factory Saat Ini',
        'Optimal_Factory': 'Pindah Ke',
        'Distance': 'Total Miles Sekarang',
        'Min_Dist': 'Total Miles Optimal',
        'Potential_Savings': 'Penghematan (mi)',
    })

    st.dataframe(
        rec_disp.style.format({
            'Total Miles Sekarang': '{:,.0f}',
            'Total Miles Optimal': '{:,.0f}',
            'Penghematan (mi)': '{:,.0f}',
            'Savings %': '{:.1f}%',
            'Units': '{:,.0f}',
        }).background_gradient(subset=['Savings %'], cmap='RdYlGn'),
        use_container_width=True, hide_index=True
    )

    total_savings = top_ineff['Potential_Savings'].sum()
    st.markdown(f"""
    <div class="success" style="color: #10B981;">
    <b>💡 Total potensi penghematan dari 10 produk teratas: {total_savings:,.0f} miles.</b><br>
    Dengan asumsi cost per mile yang masuk akal, ini bisa jadi penghematan logistik yang signifikan.
    Produk-produk di atas adalah yang paling konsisten dikirim dari factory yang jauh padahal ada
    factory lain yang lebih dekat ke customer.
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 3 — TREN PENJUALAN
# ------------------------------------------------------------
with tab3:
    st.header("📈 Tren Penjualan 2021–2024")

    monthly = (fdf.groupby('YearMonth')[['Sales', 'Gross Profit', 'Units']]
               .sum().reset_index())

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Sales Bulanan ($)",
                                        "Gross Profit Bulanan ($)",
                                        "Units Bulanan"),
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['Sales'],
                             mode='lines', fill='tozeroy', name='Sales',
                             line=dict(color='#2E86AB', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['Gross Profit'],
                             mode='lines', fill='tozeroy', name='Profit',
                             line=dict(color='#06A77D', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['Units'],
                             mode='lines', fill='tozeroy', name='Units',
                             line=dict(color='#D62246', width=2)), row=3, col=1)
    fig.update_layout(height=600, showlegend=False, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight" style="color: White;">
    Ketiga metrik bergerak bareng dan ada pola <b>seasonal Q4</b> yang kuat —
    sales selalu naik tinggi di Oktober–Desember tiap tahun. Wajar buat distributor permen karena
    Halloween, Thanksgiving, dan Christmas semua jatuh di Q4. Trend keseluruhan juga naik dari
    2021 ke 2024, jadi bisnisnya tumbuh.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Perbandingan Tahunan")
    yearly = fdf.groupby('Year').agg(
        Sales=('Sales', 'sum'),
        Profit=('Gross Profit', 'sum'),
        Orders=('Order ID', 'nunique')
    ).reset_index()

    cy1, cy2 = st.columns(2)
    with cy1:
        fig_y1 = px.bar(yearly, x='Year', y='Sales', text_auto='.2s',
                        color='Sales', color_continuous_scale='Blues',
                        title="Sales per Tahun")
        fig_y1.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig_y1, use_container_width=True)
    with cy2:
        fig_y2 = px.bar(yearly, x='Year', y='Orders', text_auto=True,
                        color='Orders', color_continuous_scale='Greens',
                        title="Jumlah Order per Tahun")
        fig_y2.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig_y2, use_container_width=True)


# ------------------------------------------------------------
# TAB 4 — TARGET vs AKTUAL
# ------------------------------------------------------------
with tab4:
    st.header("🎯 Target vs Aktual 2024")
    st.caption("Tab ini tidak menggunakan filter sidebar.")

    a24 = (df[df['Year'] == 2024].groupby('Division')
           .agg(Sales_2024=('Sales', 'sum'), Units_2024=('Units', 'sum')))
    a23 = (df[df['Year'] == 2023].groupby('Division')
           .agg(Sales_2023=('Sales', 'sum'), Units_2023=('Units', 'sum')))
    tgt = targets_df.set_index('Division')

    var = a24.join(tgt).join(a23)
    var['Variance_Pct']   = (var['Sales_2024'] - var['Target']) / var['Target'] * 100
    var['Volume_Change']  = (var['Units_2024'] / var['Units_2023'] - 1) * 100
    var['Avg_Price_2024'] = var['Sales_2024'] / var['Units_2024']
    var['Avg_Price_2023'] = var['Sales_2023'] / var['Units_2023']
    var['Price_Change']   = (var['Avg_Price_2024'] / var['Avg_Price_2023'] - 1) * 100
    var = var.reset_index()

    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(name='Target', x=var['Division'], y=var['Target'],
                           marker_color='#9CA3AF',
                           text=var['Target'].apply(lambda v: f"${v:,.0f}"),
                           textposition='outside'))
    fig_t.add_trace(go.Bar(name='Aktual 2024', x=var['Division'], y=var['Sales_2024'],
                           marker_color='#2E86AB',
                           text=var['Sales_2024'].apply(lambda v: f"${v:,.0f}"),
                           textposition='outside'))
    fig_t.update_layout(barmode='group', height=420,
                        title="Target vs Aktual Sales per Divisi (2024)",
                        yaxis_title="Sales ($)")
    st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("Dekomposisi Variance: Volume vs Harga")
    disp = var[['Division', 'Target', 'Sales_2024', 'Variance_Pct',
                'Volume_Change', 'Price_Change']].rename(columns={
        'Target': 'Target ($)',
        'Sales_2024': 'Aktual 2024 ($)',
        'Variance_Pct': 'Variance %',
        'Volume_Change': 'Volume Δ %',
        'Price_Change': 'Harga Δ %',
    })

    st.dataframe(
        disp.style.format({
            'Target ($)': '${:,.0f}',
            'Aktual 2024 ($)': '${:,.0f}',
            'Variance %': '{:+.1f}%',
            'Volume Δ %': '{:+.1f}%',
            'Harga Δ %': '{:+.1f}%',
        }).background_gradient(subset=['Variance %'], cmap='RdYlGn',
                               vmin=-100, vmax=100),
        use_container_width=True, hide_index=True
    )

    st.markdown("""
    <div class="success" style="color: #10B981;">
    <b>✅ Chocolate — pertumbuhan paling sehat:</b> Lewati target +60% dengan volume naik 27% dan
    harga stabil. Pertumbuhan organik murni karena permintaan naik.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
    <b>📌 Other — juga sehat:</b> Lewati target +15% dengan volume naik 43% dan harga turun tipis (-2%).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="danger" style="color: #DC2626;">
    <b>🚨 Sugar — gagal total:</b> Cuma capai <b>1% dari target</b> ($143 vs target $15.000).
    Volume cuma naik tipis (+2%) padahal harga sudah dipotong 31%. Artinya barang sudah didiskon
    dalam-dalam tapi tetap nggak laku. Ini perlu evaluasi serius — apakah produk Sugar masih relevan
    atau sudah waktunya di-discontinue?
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 5 — MARGIN & FACTORY
# ------------------------------------------------------------
with tab5:
    st.header("💰 Analisis Margin per Factory")

    fm = (fdf.groupby('Factory')
          .agg(Sales=('Sales', 'sum'),
               Cost=('Cost', 'sum'),
               Gross_Profit=('Gross Profit', 'sum'))
          .assign(Margin_Pct=lambda x: x['Gross_Profit'] / x['Sales'] * 100)
          .sort_values('Margin_Pct')
          .reset_index())

    fig_fm = px.bar(fm, x='Margin_Pct', y='Factory', orientation='h',
                    color='Margin_Pct', color_continuous_scale='RdYlGn',
                    text=fm['Margin_Pct'].apply(lambda v: f"{v:.1f}%"),
                    hover_data=['Sales', 'Gross_Profit'])
    fig_fm.update_layout(height=400, coloraxis_showscale=False,
                         xaxis_title="Gross Margin (%)",
                         title="Gross Margin per Factory")
    st.plotly_chart(fig_fm, use_container_width=True)

    st.markdown("""
    <div class="danger" style="color: #DC2626;">
    <ul>
    <li><b>Lot's O' Nuts</b> dan <b>Wicked Choccy's</b> — margin sehat (~65–69%)</li>
    <li><b>Sugar Shack</b> dan <b>Secret Factory</b> — di tengah (50–55%)</li>
    <li><b>The Other Factory</b> — cuma <b>~12% margin</b>, jauh banget di bawah yang lain</li>
    </ul>

    Selisih 50+ poin persentase tidak normal. Diantara cost structure factory ini memang jauh
    lebih mahal, atau produknya dijual dengan markup rendah. Kalau produk-produk di sini bisa
    dipindah ke factory yang lebih efisien, potensi penambahan profit-nya signifikan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col_md, col_pd = st.columns(2)

    with col_md:
        st.subheader("Margin per Division")
        mdiv = (fdf.groupby('Division')['Margin'].mean() * 100).sort_values(ascending=False).reset_index()
        mdiv.columns = ['Division', 'Margin %']
        fig_md = px.bar(mdiv, x='Division', y='Margin %',
                        color='Division', text_auto='.1f',
                        color_discrete_sequence=px.colors.sequential.Magma_r)
        fig_md.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_md, use_container_width=True)

    with col_pd:
        st.subheader("Persentase Unit Terjual per Divisi")
        div_units = fdf.groupby('Division')['Units'].sum().reset_index()
        fig_pd = px.pie(div_units, values='Units', names='Division', hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pd.update_traces(textposition='inside', textinfo='percent+label')
        fig_pd.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_pd, use_container_width=True)

    st.markdown("""
    <div class="insight">
    Di dataset ini margin aktual sama persis dengan margin price-list,
    artinya <b>tidak ada diskon yang diberikan</b> — semua produk dijual sesuai harga normal.
    Ini bagus karena angka margin yang kita lihat murni mencerminkan struktur biaya.
    Volume dominan dipegang Chocolate division (lihat pie chart).
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# TAB 6 — KORELASI
# ------------------------------------------------------------
with tab6:
    st.header("🔍 Korelasi Antar Variable")

    num_cols = ['Sales', 'Units', 'Cost', 'Gross Profit',
                'Distance', 'Shipping Duration', 'Margin']
    corr = fdf[num_cols].corr()

    fig_corr = px.imshow(corr, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    <div class="insight">
    <ul>
    <li><b>Sales ↔ Cost</b> dan <b>Sales ↔ Gross Profit</b> mendekati 1 — wajar, mereka secara
        matematis terkait.</li>
    <li><b>Sales ↔ Margin</b> — kalau negatif berarti order besar punya margin lebih rendah
        (volume discount). Di data ini mendekati nol, jadi ukuran order tidak ngaruh ke margin.</li>
    <li><b>Distance ↔ Shipping Duration</b> — harusnya korelasi positif lemah. Kalau lemah berarti
        durasi pengiriman lebih dipengaruhi mode pengiriman, bukan jarak fisik.</li>
    <li><b>Distance ↔ Sales / Margin</b> — kalau mendekati nol berarti jarak nggak ngaruh ke
        profitabilitas, jadi geografi bukan faktor cost yang signifikan dalam dataset ini.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Scatter: Distance vs Margin")
    sample = fdf.dropna(subset=['Distance']).sample(min(2000, len(fdf)), random_state=42)
    fig_sc = px.scatter(sample, x='Distance', y='Margin',
                        color='Division', opacity=0.5,
                        hover_data=['Product Name', 'Factory'],
                        title="Apakah jarak ngaruh ke margin?")
    fig_sc.update_layout(height=450)
    st.plotly_chart(fig_sc, use_container_width=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; font-size: 13px;'>
🍬 Customer Shipping Analysis | Kelompok 4  Study Case 4<br>
</div>
""", unsafe_allow_html=True)
