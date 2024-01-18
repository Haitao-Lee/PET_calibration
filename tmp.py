void MyGraphicView::slot_reset()
{
    QRectF rectItem = scene()->itemsBoundingRect();
    QRectF rectView = m_view->rect();
    qreal ratioView = rectView.height() / rectView.width();
    qreal ratioItem = rectItem.height() / rectItem.width();
    if (ratioView > ratioItem)
    {
        rectItem.moveTop(rectItem.width()*ratioView - rectItem.height());
        rectItem.setHeight(rectItem.width()*ratioView);

        rectItem.setWidth(rectItem.width() * 1.2);
        rectItem.setHeight(rectItem.height() * 1.2);
    }
    else
    {
        rectItem.moveLeft(rectItem.height()/ratioView - rectItem.width());
        rectItem.setWidth(rectItem.height()/ratioView);

        rectItem.setWidth(rectItem.width() * 1.2);
        rectItem.setHeight(rectItem.height() * 1.2);
    }

    m_view->fitInView(rectItem, Qt::KeepAspectRatio);
}
